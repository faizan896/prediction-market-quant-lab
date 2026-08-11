"""Synthetic market generator.

Used for tests and for sanity-checking the engine: if a strategy cannot make
money on data generated with a known edge, the bug is in the engine, not the
strategy. Conversely, a strategy that prints money on pure martingale paths is
a red flag for lookahead bias.

Generation model
----------------
Each market draws a true probability p*. Its price is a martingale that
converges to the realized outcome as expiry approaches (a Brownian bridge in
logit space), plus microstructure noise and an optional systematic
mispricing (`bias_fn`) for the engine to discover.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import numpy as np

from quantlab.types import Market, PricePoint, Venue


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class SyntheticClient:
    venue = Venue.SYNTHETIC

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        n_markets: int = 50,
        n_steps: int = 120,
        vol: float = 0.25,
        spread: float = 0.02,
        bias_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        start: datetime | None = None,
        step: timedelta = timedelta(hours=1),
    ) -> tuple[list[Market], list[PricePoint]]:
        start = start or datetime.now(timezone.utc) - n_steps * step
        markets: list[Market] = []
        points: list[PricePoint] = []

        for i in range(n_markets):
            p_true = float(self.rng.beta(1.4, 1.4))
            outcome = float(self.rng.random() < p_true)

            # Brownian bridge in logit space from p_true to the outcome.
            t = np.arange(n_steps) / max(n_steps - 1, 1)
            noise = self.rng.normal(0, vol, n_steps).cumsum()
            noise = noise - t * noise[-1]  # pin the endpoint
            # Converge to 0.97 / 0.03, not 0.999 / 0.001. Real markets keep a
            # residual tail right up to expiry — settlement risk, resolution
            # ambiguity, and people holding lottery tickets. Pinning to the
            # extremes makes tail strategies untestable.
            target = _logit(0.97 if outcome else 0.03)
            path = _logit(p_true) * (1 - t) + target * t + noise
            fair = _sigmoid(path)

            price = fair.copy()
            if bias_fn is not None:
                price = np.clip(bias_fn(price), 0.01, 0.99)

            mid = "-".join(["syn", f"{i:04d}"])
            for k in range(n_steps):
                ts = start + k * step
                half = spread / 2
                points.append(
                    PricePoint(
                        venue=Venue.SYNTHETIC,
                        market_id=mid,
                        ts=ts,
                        price=float(price[k]),
                        volume=float(self.rng.gamma(2.0, 500.0)),
                        yes_bid=float(max(0.001, price[k] - half)),
                        yes_ask=float(min(0.999, price[k] + half)),
                    )
                )

            markets.append(
                Market(
                    venue=Venue.SYNTHETIC,
                    market_id=mid,
                    event_id=f"syn-event-{i // 5:03d}",
                    title=f"Synthetic market {i} (p*={p_true:.2f})",
                    slug=mid,
                    yes_bid=float(price[-1] - spread / 2),
                    yes_ask=float(price[-1] + spread / 2),
                    last_price=float(price[-1]),
                    volume=float(self.rng.gamma(3.0, 20_000.0)),
                    liquidity=float(self.rng.gamma(2.0, 5_000.0)),
                    open_time=start,
                    close_time=start + n_steps * step,
                    status="closed",
                    resolved=True,
                    resolution=outcome,
                    min_tick=0.01,
                    fee_model="none",
                    raw={"p_true": p_true},
                )
            )
        return markets, points

    @staticmethod
    def longshot_bias(strength: float = 0.03) -> Callable[[np.ndarray], np.ndarray]:
        """Plant the real-world favourite-longshot bias in the price path.

        Longshots trade *above* their fair probability and favourites *below*,
        which is the direction actually observed in betting and prediction
        markets. Backtesting `LongshotBias` on data generated with this is the
        pipeline's end-to-end check: a known, planted edge must be recoverable.
        """
        def fn(p: np.ndarray) -> np.ndarray:
            return p + strength * np.sin(np.pi * p) * (1 - 2 * p)
        return fn

    def list_markets(self, limit: int = 200, **_kw) -> list[Market]:
        return self.generate(n_markets=limit)[0]

    def price_history(self, market_id: str, **_kw) -> list[PricePoint]:
        _, pts = self.generate(n_markets=1)
        return [PricePoint(p.venue, market_id, p.ts, p.price, p.volume, p.yes_bid, p.yes_ask)
                for p in pts]

    def order_book(self, market_id: str):  # pragma: no cover - not meaningful here
        return None
