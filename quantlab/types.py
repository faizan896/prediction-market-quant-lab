"""Venue-agnostic domain types.

Every venue client normalizes into these so that strategies, the backtester and
the scanner never know or care which exchange the data came from.

Price convention: everything is the probability of YES, in [0, 1].
Kalshi cents and Polymarket decimals are both converted on ingest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Venue(str, Enum):
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"
    SYNTHETIC = "synthetic"


class Side(str, Enum):
    YES = "yes"
    NO = "no"


@dataclass(slots=True)
class Market:
    """A single binary contract, normalized across venues."""

    venue: Venue
    market_id: str  # venue-native id (Polymarket token id / Kalshi ticker)
    event_id: str  # grouping key (Polymarket event id / Kalshi event ticker)
    title: str
    slug: str = ""
    # Trading state
    yes_bid: float | None = None
    yes_ask: float | None = None
    last_price: float | None = None
    # Liquidity
    volume: float = 0.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    open_interest: float = 0.0
    # Lifecycle
    open_time: datetime | None = None
    close_time: datetime | None = None
    status: str = "open"
    resolved: bool = False
    resolution: float | None = None  # 1.0 YES, 0.0 NO, None unresolved
    # Fees, tick size
    min_tick: float = 0.01
    fee_model: str = "none"  # "kalshi" | "polymarket" | "none"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.venue.value}:{self.market_id}"

    @property
    def mid(self) -> float | None:
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2.0
        return self.last_price

    @property
    def spread(self) -> float | None:
        if self.yes_bid is not None and self.yes_ask is not None:
            return self.yes_ask - self.yes_bid
        return None


@dataclass(slots=True)
class PricePoint:
    """One observation in a market's price history."""

    venue: Venue
    market_id: str
    ts: datetime
    price: float
    volume: float = 0.0
    yes_bid: float | None = None
    yes_ask: float | None = None


@dataclass(slots=True)
class BookLevel:
    price: float
    size: float


@dataclass(slots=True)
class OrderBook:
    venue: Venue
    market_id: str
    ts: datetime
    bids: list[BookLevel] = field(default_factory=list)  # descending price
    asks: list[BookLevel] = field(default_factory=list)  # ascending price

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    def microprice(self, depth: int = 3) -> float | None:
        """Size-weighted price. Leans toward the side with less resting size,
        because that is the side the book is about to walk through."""
        bids, asks = self.bids[:depth], self.asks[:depth]
        if not bids or not asks:
            return None
        bid_sz = sum(b.size for b in bids)
        ask_sz = sum(a.size for a in asks)
        if bid_sz + ask_sz == 0:
            return (bids[0].price + asks[0].price) / 2
        return (bids[0].price * ask_sz + asks[0].price * bid_sz) / (bid_sz + ask_sz)

    def depth_within(self, ticks: float) -> tuple[float, float]:
        """Resting size within `ticks` of the touch, (bid_size, ask_size)."""
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return (0.0, 0.0)
        b = sum(l.size for l in self.bids if l.price >= bb - ticks)
        a = sum(l.size for l in self.asks if l.price <= ba + ticks)
        return (b, a)


@dataclass(slots=True)
class FairEstimate:
    """One signal's opinion about a market's true probability."""

    source: str
    prob: float
    confidence: float  # 0..1, used as the pooling weight
    note: str = ""
