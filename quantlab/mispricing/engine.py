"""The Mispricing Engine.

    Market -> Current Price -> Fair Value -> Edge -> Confidence -> Expected Value

Pools every signal's opinion into a single fair probability, subtracts the cost
of actually trading it, and ranks what's left.

Pooling
-------
Opinions are combined in **logit space**, weighted by (signal weight x
confidence). Logit pooling is the right choice here rather than averaging
probabilities: averaging 0.02 and 0.20 gives 0.11, which badly overstates the
longshot; in logit space you get 0.063, which is closer to how a Bayesian
updates. It also can't produce a fair value outside [0,1] and it treats the
odds ratio, not the raw probability, as the thing being averaged.

Confidence
----------
Reported confidence combines agreement (do the signals concur?) with evidence
(how much non-market weight is there?). Two signals agreeing on 0.64 is a very
different situation from one signal shouting 0.64 while the market says 0.57,
and the ranking must reflect that.

What "edge" is not
------------------
`edge` is the raw gap between fair value and price. `net_ev` is the number that
matters: edge minus spread, fees and slippage, per dollar deployed. A 4-point
edge on a market with a 3-cent spread is not a trade.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quantlab.backtest.frictions import PRESETS, FrictionModel
from quantlab.backtest.sizing import SizingConfig, kelly_fraction, position_notional
from quantlab.mispricing.signals import Signal, default_signals, logit, sigmoid
from quantlab.types import FairEstimate, Market, OrderBook


@dataclass
class Opportunity:
    key: str
    venue: str
    title: str
    price: float
    fair_value: float
    edge: float  # fair - price, in YES terms (signed by side below)
    side: str  # "YES" or "NO"
    confidence: float
    gross_ev: float  # per dollar deployed, before costs
    net_ev: float  # per dollar deployed, after spread/fees/slippage
    kelly_fraction: float
    suggested_notional: float
    spread: float
    liquidity: float
    volume_24h: float
    hours_to_close: float | None
    contributors: dict[str, float]
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class MispricingEngine:
    """Scan a universe of markets and rank the mispriced ones."""

    def __init__(
        self,
        signals: list[Signal] | None = None,
        weights: dict[str, float] | None = None,
        frictions: FrictionModel | None = None,
        sizing: SizingConfig | None = None,
        bankroll: float = 100_000.0,
        market_anchor: float = 0.30,
    ):
        """`market_anchor` is a *second* pull toward the market price, applied
        after pooling. It is deliberately modest, because `MarketPrior` already
        sits in the pool at weight 3.0 and does most of the anchoring; setting
        this high as well double-counts and crushes a legitimate 7-point view
        down to under a point.

        Keep it non-zero, though. Overriding a liquid market by 20 points is
        almost always a bug in your data, not an edge, and this is the last line
        of defence against a broken signal sizing a large position."""
        self.signals = signals or default_signals()
        self.weights = weights or {s.name: s.default_weight for s in self.signals}
        self.frictions = frictions or PRESETS["conservative"]
        self.sizing = sizing or SizingConfig()
        self.bankroll = bankroll
        self.market_anchor = market_anchor

    # ------------------------------------------------------------------ pooling
    def combine(self, price: float, estimates: list[FairEstimate]) -> tuple[float, float, dict]:
        """Weighted logit pool. Returns (fair_value, confidence, contributions)."""
        usable = [e for e in estimates if e is not None and np.isfinite(e.prob)]
        if not usable:
            return price, 0.0, {}

        contrib: dict[str, float] = {}
        num = 0.0
        den = 0.0
        for e in usable:
            w = self.weights.get(e.source, 1.0) * max(e.confidence, 0.0)
            if w <= 0:
                continue
            num += w * logit(e.prob)
            den += w
            contrib[e.source] = round(w, 4)

        if den <= 0:
            return price, 0.0, {}

        pooled = sigmoid(num / den)

        # Anchor toward the market price.
        fair = sigmoid(self.market_anchor * logit(price) + (1 - self.market_anchor) * logit(pooled))

        # --- confidence -----------------------------------------------------
        probs = np.array([e.prob for e in usable])
        ws = np.array([self.weights.get(e.source, 1.0) * e.confidence for e in usable])
        ws = ws / ws.sum() if ws.sum() > 0 else np.ones_like(ws) / len(ws)
        dispersion = float(np.sqrt(np.sum(ws * (probs - fair) ** 2)))
        agreement = float(np.clip(1.0 - dispersion * 6.0, 0.0, 1.0))

        non_market = sum(v for k, v in contrib.items() if k != "market")
        evidence = float(np.clip(non_market / 2.0, 0.0, 1.0))

        confidence = float(np.clip(0.35 * evidence + 0.65 * agreement, 0.0, 1.0))
        return fair, confidence, contrib

    # -------------------------------------------------------------------- scan
    def evaluate(
        self,
        market: Market,
        reference: Market | None = None,
        book: OrderBook | None = None,
        history: list | None = None,
    ) -> Opportunity | None:
        price = market.mid
        if price is None or not (0.005 < price < 0.995):
            return None

        estimates = [
            s.estimate(market, reference=reference, book=book, history=history)
            for s in self.signals
        ]
        fair, confidence, contrib = self.combine(price, estimates)

        raw_edge = fair - price
        side = "YES" if raw_edge > 0 else "NO"
        side_sign = 1.0 if raw_edge > 0 else -1.0

        # --- expected value, per dollar deployed ----------------------------
        # Long YES:  cost b = price, payoff 1 with prob fair.  EV/$ = (fair-b)/b
        # Long NO:   cost 1-price, payoff 1 with prob 1-fair.  EV/$ = (b-fair)/(1-b)
        basis = price if side_sign > 0 else 1.0 - price
        gross_ev = float(abs(raw_edge) / max(basis, 1e-6))

        # --- sizing (needed before costs: fees are size-dependent) -----------
        kf = float(kelly_fraction(np.array([fair]), np.array([price]), np.array([side_sign]))[0])
        notional = float(position_notional(
            self.sizing, self.bankroll, np.array([fair]), np.array([price]),
            np.array([side_sign]),
            liquidity=np.array([market.liquidity]) if market.liquidity else None,
        )[0])
        # Discount size by how much we believe the estimate.
        notional *= confidence

        # --- costs, per dollar deployed --------------------------------------
        # Priced at the order size we would actually send, then divided back
        # down. Kalshi rounds its fee up to the whole cent *per order*, so
        # normalizing to a $1 unit first would inflate the fee by orders of
        # magnitude on any realistic clip.
        spread = market.spread if market.spread is not None else \
            self.frictions.spread_ticks * self.frictions.tick * 2
        fr = FrictionModel(
            spread_ticks=self.frictions.spread_ticks,
            tick=market.min_tick or 0.01,
            slippage_bps=self.frictions.slippage_bps,
            impact_coef=self.frictions.impact_coef,
            fee_model=market.fee_model if market.fee_model != "none" else self.frictions.fee_model,
            fee_rate=self.frictions.fee_rate,
        )
        ref_notional = max(notional, self.sizing.min_notional, 1_000.0)
        contracts = np.array([ref_notional / max(basis, 1e-6)])
        entry = np.array([price + side_sign * spread / 2])
        cost_dollars = float(fr.round_trip_cost(
            entry, np.array([float(side_sign > 0)]), contracts,
            np.array([market.liquidity or np.nan]),
        )[0])
        cost = cost_dollars / ref_notional
        cost += (spread / 2) / max(basis, 1e-6)  # crossing the spread on entry
        net_ev = gross_ev - cost

        hours = None
        if market.close_time:
            hours = max((market.close_time - datetime.now(timezone.utc)).total_seconds() / 3600.0, 0.0)

        return Opportunity(
            key=market.key,
            venue=market.venue.value,
            title=market.title,
            price=round(price, 4),
            fair_value=round(fair, 4),
            edge=round(abs(raw_edge), 4),
            side=side,
            confidence=round(confidence, 3),
            gross_ev=round(gross_ev, 4),
            net_ev=round(net_ev, 4),
            kelly_fraction=round(kf, 4),
            suggested_notional=round(notional, 2),
            spread=round(spread, 4),
            liquidity=market.liquidity,
            volume_24h=market.volume_24h,
            hours_to_close=None if hours is None else round(hours, 1),
            contributors={k: round(v, 3) for k, v in contrib.items()},
            notes=" | ".join(e.note for e in estimates if e and e.note)[:200],
        )

    def scan(
        self,
        markets: list[Market],
        references: dict[str, Market] | None = None,
        books: dict[str, OrderBook] | None = None,
        min_edge: float = 0.02,
        min_net_ev: float = 0.0,
        min_confidence: float = 0.0,
        min_liquidity: float = 0.0,
    ) -> pd.DataFrame:
        references = references or {}
        books = books or {}
        rows: list[dict] = []
        for m in markets:
            if m.liquidity and m.liquidity < min_liquidity:
                continue
            opp = self.evaluate(m, reference=references.get(m.key), book=books.get(m.key))
            if opp is None:
                continue
            if opp.edge < min_edge or opp.net_ev < min_net_ev or opp.confidence < min_confidence:
                continue
            rows.append(opp.to_dict())

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # Rank by risk-adjusted expected value, not raw edge. Ranking by edge
        # puts every illiquid 1-cent contract at the top of the screen.
        df["score"] = df["net_ev"] * df["confidence"]
        return df.sort_values("score", ascending=False).reset_index(drop=True)
