"""Fair-value signals.

Each signal answers one question: given what I know, what is the true
probability of YES? It returns a `FairEstimate` with a confidence, and the
combiner pools them.

Design rule: a signal must be able to say "I don't know" (confidence 0). A
signal that always has an opinion will drown out the ones that only speak when
they have something to say, which is where the actual edge lives.

Adding a new source (polls, weather models, sportsbook lines, economic
releases) means writing one class here. Nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone

import numpy as np

from quantlab.types import FairEstimate, Market, OrderBook


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class Signal(ABC):
    name: str = "signal"
    default_weight: float = 1.0

    @abstractmethod
    def estimate(self, market: Market, **ctx) -> FairEstimate | None:
        """Return an opinion, or None to abstain."""


class MarketPrior(Signal):
    """The market price itself.

    Always included, and it should usually dominate. The market is the
    aggregate of everyone with money on the line; your model needs a specific
    reason to disagree, not a general one. This signal is what keeps the
    scanner from confidently flagging every contract as mispriced.
    """

    name = "market"
    default_weight = 3.0

    def estimate(self, market: Market, **ctx) -> FairEstimate | None:
        mid = market.mid
        if mid is None:
            return None
        # Wide spreads mean the price is a weak signal about consensus.
        spread = market.spread or 0.05
        conf = float(np.clip(1.0 - spread * 8, 0.2, 1.0))
        return FairEstimate(self.name, mid, conf, f"mid={mid:.3f} spread={spread:.3f}")


class CrossVenue(Signal):
    """The same contract's price on the other exchange.

    The most reliable non-market signal available, because it is another set of
    traders with money at risk. A persistent gap is either an inefficiency or a
    resolution-criteria difference — check the rules text before trusting it.
    """

    name = "cross_venue"
    default_weight = 2.0

    def estimate(self, market: Market, reference: Market | None = None, **ctx):
        if reference is None or reference.mid is None:
            return None
        ref_mid = reference.mid
        liq = reference.liquidity or reference.volume_24h or 0.0
        conf = float(np.clip(np.log1p(liq) / np.log1p(100_000), 0.15, 0.9))
        return FairEstimate(
            self.name, ref_mid, conf,
            f"{reference.venue.value}@{ref_mid:.3f}",
        )


class OrderBookImbalance(Signal):
    """Microprice from resting depth.

    When the ask side is thin relative to the bid, the next print is more likely
    to be higher. This is a genuine short-horizon edge, but it decays in minutes
    — useful for execution timing, weak as a standalone fair value.
    """

    name = "book_imbalance"
    default_weight = 0.8

    def __init__(self, depth: int = 3):
        self.depth = depth

    def estimate(self, market: Market, book: OrderBook | None = None, **ctx):
        if book is None:
            return None
        micro = book.microprice(self.depth)
        if micro is None:
            return None
        bid_sz, ask_sz = book.depth_within(0.02)
        total = bid_sz + ask_sz
        if total <= 0:
            return None
        imbalance = abs(bid_sz - ask_sz) / total
        conf = float(np.clip(imbalance * 0.6, 0.05, 0.5))
        return FairEstimate(self.name, micro, conf, f"micro={micro:.3f} imb={imbalance:.2f}")


class LongshotCorrection(Signal):
    """Systematic correction for the favourite-longshot bias.

    Applies a logit-space stretch that pushes cheap contracts cheaper and
    expensive ones dearer. Strength should be *fitted on your own resolved
    markets*, not taken from this default — the magnitude varies a lot by
    category (sports markets are near-efficient; political longshots are not).
    """

    name = "longshot"
    default_weight = 1.0

    def __init__(self, strength: float = 0.30, min_hours: float = 24.0):
        self.strength = strength
        self.min_hours = min_hours

    def estimate(self, market: Market, **ctx):
        mid = market.mid
        if mid is None or not (0.005 < mid < 0.995):
            return None
        hours = _hours_to_close(market)
        if hours is None:
            decay = 0.5
        else:
            decay = float(np.clip(hours / (hours + self.min_hours), 0.0, 1.0))
        fair = sigmoid(logit(mid) * (1 + self.strength * decay))
        # Only worth listening to away from the middle.
        conf = float(np.clip(abs(mid - 0.5) * 1.2, 0.05, 0.6)) * decay
        return FairEstimate(self.name, fair, conf, f"decay={decay:.2f}")


class ExternalProbability(Signal):
    """Bring your own model.

    This is where polls, weather forecasts, sportsbook lines, an econometric
    model, or an LLM's structured estimate plug in. Supply a dict of
    {market_key: (probability, confidence)}; everything downstream — pooling,
    edge, Kelly sizing, backtesting — works unchanged.

    The whole architecture exists to make this class easy to write.
    """

    name = "external"
    default_weight = 2.5

    def __init__(self, table: dict[str, tuple[float, float]] | None = None, source: str = "external"):
        self.table = table or {}
        self.source = source

    def estimate(self, market: Market, **ctx):
        hit = self.table.get(market.key) or self.table.get(market.market_id)
        if hit is None:
            return None
        prob, conf = hit
        return FairEstimate(self.source, float(prob), float(np.clip(conf, 0, 1)), "external model")


class VolumeWeightedDrift(Signal):
    """Recent volume-weighted price drift, as a momentum tilt on fair value."""

    name = "drift"
    default_weight = 0.7

    def __init__(self, max_tilt: float = 0.04):
        self.max_tilt = max_tilt

    def estimate(self, market: Market, history: list | None = None, **ctx):
        if not history or len(history) < 6:
            return None
        prices = np.array([h.price for h in history[-24:]], dtype=float)
        vols = np.array([max(h.volume, 1e-6) for h in history[-24:]], dtype=float)
        vwap = float(np.average(prices, weights=vols))
        drift = prices[-1] - vwap
        fair = float(np.clip(prices[-1] + np.clip(drift, -self.max_tilt, self.max_tilt), 0.01, 0.99))
        conf = float(np.clip(abs(drift) * 10, 0.05, 0.45))
        return FairEstimate(self.name, fair, conf, f"vwap={vwap:.3f} drift={drift:+.3f}")


def _hours_to_close(market: Market) -> float | None:
    if market.close_time is None:
        return None
    now = datetime.now(timezone.utc)
    return max((market.close_time - now).total_seconds() / 3600.0, 0.0)


def default_signals() -> list[Signal]:
    return [MarketPrior(), CrossVenue(), OrderBookImbalance(), LongshotCorrection(), VolumeWeightedDrift()]
