"""Trading frictions: spread, slippage, fees.

This module is the difference between a backtest that looks like the leaderboard
in the README and one you can actually trade. Most prediction-market "edges"
smaller than ~3 cents are entirely consumed by what happens in here.

All functions are vectorized over numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FrictionModel:
    """Configurable cost model.

    spread_ticks
        Half-spread paid on entry and exit, in ticks, when explicit bid/ask are
        unavailable. With 1c ticks, 1.0 means you cross a 2c-wide market.
    slippage_bps
        Fixed cost per side in basis points of notional.
    impact_coef
        Square-root market impact: cost = impact_coef * sqrt(size / liquidity),
        as a fraction of notional. Prediction-market books are thin, so this
        term dominates on large clips. Note it stacks on top of the sizing
        layer's `max_liquidity_share` cap; with that cap at 10%, an
        impact_coef of 0.15 costs at most ~4.7% per side.
    fee_model
        "kalshi" | "polymarket" | "none".
    fee_rate
        Overrides the venue default rate when set.
    """

    spread_ticks: float = 1.0
    tick: float = 0.01
    slippage_bps: float = 10.0
    impact_coef: float = 0.0
    fee_model: str = "none"
    fee_rate: float | None = None
    maker_fraction: float = 0.0  # share of volume filled passively (no taker fee)

    # ------------------------------------------------------------------ prices
    def entry_price(
        self,
        mid: np.ndarray,
        side: np.ndarray,
        bid: np.ndarray | None = None,
        ask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Price paid to open. side=+1 buys YES, side=-1 buys NO.

        Buying NO at price q is economically shorting YES at 1-q, so we express
        everything in YES terms: a short entry fills at the bid.
        """
        half = self.spread_ticks * self.tick
        if bid is not None and ask is not None:
            up = np.where(np.isfinite(ask), ask, mid + half)
            down = np.where(np.isfinite(bid), bid, mid - half)
        else:
            up, down = mid + half, mid - half
        px = np.where(side > 0, up, down)
        return np.clip(px, 0.001, 0.999)

    def exit_price(
        self,
        mid: np.ndarray,
        side: np.ndarray,
        bid: np.ndarray | None = None,
        ask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Price received on close — the opposite side of the book from entry."""
        return self.entry_price(mid, -side, bid, ask)

    # ------------------------------------------------------------------- costs
    def slippage(self, notional: np.ndarray, liquidity: np.ndarray | None = None) -> np.ndarray:
        cost = notional * (self.slippage_bps / 10_000.0)
        if self.impact_coef and liquidity is not None:
            liq = np.where(np.asarray(liquidity) > 0, liquidity, np.inf)
            cost = cost + notional * self.impact_coef * np.sqrt(notional / liq)
        return cost

    def fees(self, price: np.ndarray, contracts: np.ndarray) -> np.ndarray:
        """Per-side trading fee in dollars.

        Kalshi:      fee = ceil(0.07 * C * P * (1 - P)) in cents, rounded up
                     per order. The P*(1-P) term means fees peak at 50c and
                     vanish at the tails.
        Polymarket:  symmetric taker fee, rate * min(P, 1-P) * C, charged on
                     the proceeds side. Only applies to fee-enabled markets.
        """
        price = np.clip(np.asarray(price, dtype=float), 0.0, 1.0)
        contracts = np.asarray(contracts, dtype=float)
        taker = 1.0 - self.maker_fraction

        if self.fee_model == "kalshi":
            rate = self.fee_rate if self.fee_rate is not None else 0.07
            raw = rate * contracts * price * (1 - price)
            # Round to sub-cent precision before the ceiling, or float error
            # turns an exact $1.75 into $1.76.
            return np.ceil(np.round(raw * 100.0, 6)) / 100.0 * taker
        if self.fee_model == "polymarket":
            rate = self.fee_rate if self.fee_rate is not None else 0.05
            return rate * np.minimum(price, 1 - price) * contracts * taker
        return np.zeros_like(contracts)

    def round_trip_cost(
        self,
        entry_px: np.ndarray,
        exit_px: np.ndarray,
        contracts: np.ndarray,
        liquidity: np.ndarray | None = None,
    ) -> np.ndarray:
        """Total dollar cost of opening and closing, excluding the spread
        (which is already embedded in entry_px / exit_px)."""
        entry_notional = entry_px * contracts
        exit_notional = exit_px * contracts
        return (
            self.fees(entry_px, contracts)
            + self.fees(exit_px, contracts)
            + self.slippage(entry_notional, liquidity)
            + self.slippage(exit_notional, liquidity)
        )


# Venue presets calibrated to published fee schedules and typical top-of-book.
PRESETS: dict[str, FrictionModel] = {
    "frictionless": FrictionModel(spread_ticks=0.0, slippage_bps=0.0, fee_model="none"),
    "polymarket": FrictionModel(
        spread_ticks=1.0, tick=0.01, slippage_bps=8.0, impact_coef=0.06,
        fee_model="polymarket", fee_rate=0.05,
    ),
    "kalshi": FrictionModel(
        spread_ticks=1.0, tick=0.01, slippage_bps=5.0, impact_coef=0.08,
        fee_model="kalshi", fee_rate=0.07,
    ),
    # What you should actually assume before risking money.
    "conservative": FrictionModel(
        spread_ticks=1.5, tick=0.01, slippage_bps=20.0, impact_coef=0.15,
        fee_model="kalshi", fee_rate=0.07,
    ),
}
