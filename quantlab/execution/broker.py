"""Brokers: the boundary between a decision and a fill.

`Broker` is the only interface the trading loop knows about. `PaperBroker`
implements it against the same `FrictionModel` the backtester uses; `LiveBroker`
is the seam where real venue clients go. Because the loop is written against the
interface, promoting a strategy from paper to live is a one-line swap and
*nothing about the decision logic changes* — which is the entire point of having
a paper stage.

Sign convention matches the backtester: everything is expressed in YES terms.
    side = +1 / "YES"   buy YES, lift the ask
    side = -1 / "NO"    buy NO, which costs (1 - yes_price) per contract
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from quantlab.backtest.frictions import PRESETS, FrictionModel
from quantlab.types import Market


@dataclass(slots=True)
class OrderIntent:
    """What the strategy wants to do, before costs are known.

    Opens are sized in dollars (`notional`), because that is how a risk limit is
    expressed. Closes and settles are sized in `contracts`, because you exit the
    position you actually hold — sizing an exit in dollars would leave a dust
    remainder every time the price moved between entry and exit.
    """

    key: str
    side: str  # "YES" | "NO"
    notional: float = 0.0  # dollars to deploy (opens)
    contracts: float | None = None  # exact size (closes and settles)
    action: str = "open"  # open | close | settle
    fair_prob: float | None = None
    edge: float | None = None
    reason: str = ""
    title: str = ""

    @property
    def sign(self) -> float:
        return 1.0 if self.side.upper() == "YES" else -1.0


@dataclass(slots=True)
class Fill:
    """What actually happened, after costs."""

    fill_id: str
    key: str
    title: str
    ts: datetime
    action: str
    side: str
    price: float  # YES-terms fill price
    contracts: float
    notional: float  # cash moved (positive = cash out on open)
    fee: float
    slippage: float
    fair_prob: float | None = None
    edge: float | None = None
    realized_pnl: float = 0.0
    reason: str = ""

    def to_row(self, run_id: str) -> dict:
        return {
            "fill_id": self.fill_id, "run_id": run_id, "key": self.key,
            "title": self.title, "ts": self.ts.isoformat(), "action": self.action,
            "side": self.side, "price": self.price, "contracts": self.contracts,
            "notional": self.notional, "fee": self.fee, "slippage": self.slippage,
            "fair_prob": self.fair_prob, "edge": self.edge,
            "realized_pnl": self.realized_pnl, "reason": self.reason,
        }


class RejectedOrder(Exception):
    """Raised when an order cannot be filled. Not an error — a normal outcome."""


class Broker(ABC):
    """Everything the trading loop is allowed to know about execution."""

    @abstractmethod
    def quote(self, market: Market) -> tuple[float, float]:
        """(bid, ask) in YES terms."""

    @abstractmethod
    def execute(self, intent: OrderIntent, market: Market) -> Fill:
        """Fill an intent or raise RejectedOrder."""


@dataclass
class PaperBroker(Broker):
    """Simulated execution with the backtester's cost model.

    Deliberately pessimistic in three ways, because the failure mode of paper
    trading is flattering yourself:

    1. You always cross the spread. No assumed passive fills.
    2. Size is capped at a share of resting liquidity, and orders that would
       take more than that are *reduced*, not rejected — matching what a real
       execution algo would do.
    3. Fees are charged on both legs at the venue's real schedule.
    """

    frictions: FrictionModel = field(default_factory=lambda: PRESETS["conservative"])
    max_liquidity_share: float = 0.10
    min_notional: float = 10.0
    fills: list[Fill] = field(default_factory=list)

    def quote(self, market: Market) -> tuple[float, float]:
        bid, ask = market.yes_bid, market.yes_ask
        mid = market.mid
        if mid is None:
            raise RejectedOrder(f"{market.key}: no price available")
        half = self.frictions.spread_ticks * (market.min_tick or 0.01)
        return (
            bid if bid is not None else max(0.001, mid - half),
            ask if ask is not None else min(0.999, mid + half),
        )

    def execute(self, intent: OrderIntent, market: Market) -> Fill:
        bid, ask = self.quote(market)
        sign = intent.sign

        # Settlement is not a trade: no spread, no fee, fills at the outcome.
        if intent.action == "settle":
            if market.resolution is None:
                raise RejectedOrder(f"{market.key}: not resolved, cannot settle")
            if not intent.contracts:
                raise RejectedOrder(f"{market.key}: settle requires a contract count")
            return self._make_fill(
                intent, market, float(market.resolution), intent.contracts, 0.0, 0.0
            )

        # Opening buys; closing sells. A close of a YES position hits the bid.
        opening = intent.action == "open"
        lift = (sign > 0) if opening else (sign < 0)
        price = ask if lift else bid
        price = float(np.clip(price, 0.001, 0.999))

        # Cost basis per contract: p for the YES leg, 1-p for the NO leg.
        basis = price if sign > 0 else 1.0 - price
        basis = max(basis, 1e-4)

        if not opening:
            if not intent.contracts:
                raise RejectedOrder(f"{market.key}: close requires a contract count")
            contracts = intent.contracts
            notional = contracts * basis
        else:
            notional = intent.notional
            if market.liquidity and market.liquidity > 0:
                notional = min(notional, market.liquidity * self.max_liquidity_share)
            if notional < self.min_notional:
                raise RejectedOrder(
                    f"{market.key}: ${notional:,.2f} below minimum "
                    f"${self.min_notional:,.2f}"
                )
            contracts = notional / basis
        fee = float(self.frictions.fees(np.array([price]), np.array([contracts]))[0])
        slip = float(self.frictions.slippage(
            np.array([notional]),
            np.array([market.liquidity]) if market.liquidity else None,
        )[0])
        return self._make_fill(intent, market, price, contracts, fee, slip)

    def _make_fill(
        self, intent: OrderIntent, market: Market,
        price: float, contracts: float, fee: float, slippage: float,
    ) -> Fill:
        sign = intent.sign
        basis = price if sign > 0 else 1.0 - price
        fill = Fill(
            fill_id=uuid.uuid4().hex[:16],
            key=intent.key,
            title=intent.title or market.title,
            ts=datetime.now(timezone.utc),
            action=intent.action,
            side=intent.side.upper(),
            price=round(price, 6),
            contracts=round(contracts, 4),
            notional=round(contracts * max(basis, 0.0), 4),
            fee=round(fee, 4),
            slippage=round(slippage, 4),
            fair_prob=intent.fair_prob,
            edge=intent.edge,
            reason=intent.reason,
        )
        self.fills.append(fill)
        return fill


class LiveBroker(Broker):
    """Seam for real order placement. Intentionally not implemented.

    Polymarket: derive API credentials from a Polygon wallet key via
    `py-clob-client`, hold USDC on Polygon, set token allowances, then post
    signed orders to the CLOB.

    Kalshi: RSA-PSS request signing with a key pair generated in your account
    settings; orders go to POST /trade-api/v2/portfolio/orders.

    Both are straightforward. The reason this raises instead of shipping half
    done is that an execution bug here loses real money, whereas every other
    bug in this repo only produces a wrong number. Implement it deliberately,
    with its own tests, after the paper record justifies it.
    """

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "LiveBroker is not implemented. Run PaperBroker until you have at "
            "least 200 settled paper trades and a calibration curve you trust."
        )

    def quote(self, market: Market) -> tuple[float, float]:  # pragma: no cover
        raise NotImplementedError

    def execute(self, intent: OrderIntent, market: Market) -> Fill:  # pragma: no cover
        raise NotImplementedError
